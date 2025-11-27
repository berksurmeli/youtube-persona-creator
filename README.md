# YouTube Persona Creator – RLHF Training Stack

This repository contains a complete local workflow for **collecting RLHF-style feedback** and **fine‑tuning a local LLM** to generate content in different _personas_.

There are two key models involved:

- **Generation model (for data collection):**  
  A local `llama-3.1-13b-instruct` model running in **LM Studio** as a server. This is what Rails calls to generate candidate outputs for ranking.
- **Training base model (for fine-tuning):**  
  A Hugging Face model such as **`meta-llama/Meta-Llama-3-8B-Instruct`**, which you fine‑tune using the collected feedback (SFT, DPO, Reward Model). This model is _separate_ from the LM Studio GGUF model.

The project is split into two parts:

- `rails-app/` – A Rails web app for:

  - Defining **Personas**
  - Creating **Topics** under Personas
  - Generating **multiple model outputs** per Topic using LM Studio
  - **Ranking** outputs (Best / Middle / Worst)
  - Exporting datasets for SFT, DPO, and Reward Model training

- `model-training/` – A Python project for:
  - **Supervised Fine‑Tuning (SFT)**
  - **Direct Preference Optimization (DPO)**
  - **Reward Model (RM) training**
  - **Merging LoRA adapters** into a full HF model
  - Converting merged models to **GGUF** for use in LM Studio

> ⚠️ This stack is designed for **local experimentation**. It assumes you have a Mac with Apple Silicon (M1/M2/M3) and Python + Ruby ready. It should also work on Linux with a few adjustments (CUDA instead of MPS, bitsandbytes, etc.).

---

## Project Structure

```text
youtube-persona-creator/
├── rails-app/              # Rails 7 app for RLHF data collection
│   ├── app/
│   │   ├── models/
│   │   │   ├── persona.rb
│   │   │   ├── topic.rb
│   │   │   ├── generated_content.rb
│   │   │   └── ranking.rb
│   │   ├── controllers/
│   │   │   ├── personas_controller.rb
│   │   │   ├── topics_controller.rb
│   │   │   └── generated_contents_controller.rb
│   │   └── views/
│   │       ├── personas/
│   │       ├── topics/
│   │       └── ...
│   ├── config/
│   ├── db/
│   ├── export_sft.jsonl
│   ├── export_dpo.jsonl
│   └── export_rm.jsonl
│
└── model-training/         # Python training pipeline
    ├── requirements.txt
    ├── train_sft.py
    ├── train_dpo.py
    ├── train_rm.py
    ├── merge_lora.py
    └── (optional) gguf models output
```

---

## 1. Rails App – Data Collection & Personas

### 1.1. Domain Models

#### Persona

Each **Persona** describes a style / voice / role you want the model to adopt.

```ruby
class Persona < ApplicationRecord
  has_many :topics, dependent: :destroy

  validates :name, presence: true
  validates :description, presence: true
end
```

Examples:

- “Hyper-enthusiastic YouTube finance guru”
- “Calm, technical AI engineer explaining ML concepts”
- “Storytelling movie critic”

#### Topic

Each **Topic** belongs to a Persona and represents a concrete content request (prompt) that you’ll generate and rank outputs for.

Example:  
“Write a YouTube video script about why most people fail at building habits.”

Typical fields:

- `persona_id` – FK to Persona
- `title` – short human-friendly name
- `description` – the actual prompt sent to the LLM

`Topic` also provides a helper to embed persona context into training data:

```ruby
class Topic < ApplicationRecord
  belongs_to :persona
  has_many :generated_contents, dependent: :destroy
  has_one :ranking, dependent: :destroy

  validates :title, :description, presence: true

  def full_prompt_for_persona
    <<~PROMPT
    You are the following persona:

    #{persona.name}
    #{persona.description}

    Generate content for this request:
    #{description}
    PROMPT
  end
end
```

This method is used when **exporting datasets** so the training data always includes persona context.

#### GeneratedContent

Each **GeneratedContent** is one LLM output for a Topic.

- `topic_id` – FK
- `content` – the text returned from LLM
- `variant_index` – 0 / 1 / 2 (etc.) to distinguish variants for ranking

You usually generate **3 variants** per topic to support multi-level ranking (Best/Middle/Worst).

#### Ranking

Stores the **relative ordering** of the variants for a Topic.

- `topic_id`
- `best_id` – `GeneratedContent` ID
- `middle_id` – `GeneratedContent` ID
- `worst_id` – `GeneratedContent` ID

This is what drives SFT, DPO, and Reward Model datasets.

---

### 1.2. Controllers (High Level)

#### PersonasController

Responsible for:

- Listing Personas
- Creating / editing Personas
- Showing Persona details and its Topics

#### TopicsController

Key actions:

```ruby
class TopicsController < ApplicationController
  def index
    @topics = Topic.includes(:persona).all
  end

  def show
    @topic = Topic.find(params[:id])
    @contents = @topic.generated_contents.order(:variant_index)
  end

  def new
    @topic = Topic.new
  end

  def create
    @topic = Topic.new(topic_params)
    if @topic.save
      redirect_to @topic, notice: "Topic created!"
    else
      render :new
    end
  end

  # GET /topics/:id/rank
  def rank
    @topic = Topic.find(params[:id])
    @contents = @topic.generated_contents.order(:variant_index)
  end

  # POST /topics/:id/submit_rank
  def submit_rank
    topic = Topic.find(params[:id])
    ranks = params[:ranking] # { "content_id" => "1|2|3" }

    content_ids_by_rank = ranks.sort_by { |_id, rank| rank.to_i }

    best_id   = content_ids_by_rank[0][0]
    middle_id = content_ids_by_rank[1][0]
    worst_id  = content_ids_by_rank[2][0]

    Ranking.create!(
      topic: topic,
      best_id: best_id,
      middle_id: middle_id,
      worst_id: worst_id
    )

    redirect_to topic_path(topic), notice: "Ranking saved!"
  end

  private

  def topic_params
    params.require(:topic).permit(:persona_id, :title, :description)
  end
end
```

#### GeneratedContentsController

Responsible for calling the **local LLM via LM Studio** and storing variants.

```ruby
class GeneratedContentsController < ApplicationController
  VARIANTS = 3  # how many outputs to generate per topic

  def create
    topic = Topic.find(params[:topic_id])

    VARIANTS.times do |i|
      result = LlmClient.generate(topic.description, persona: topic.persona)
      topic.generated_contents.create!(
        content: result,
        variant_index: i
      )
    end

    redirect_to topic_path(topic), notice: "Generated #{VARIANTS} outputs!"
  end
end
```

Here:

- The **generation prompt** sent to LM Studio is the _topic description_ plus a system persona message.
- The **training instruction** (in exporters) uses `full_prompt_for_persona`, which inlines persona + topic into a single string. So persona influences both generation and training.

---

### 1.3. LLM Client (LM Studio Integration)

LM Studio runs a local server (OpenAI-compatible) e.g. at:

- `http://localhost:1234/v1/chat/completions`
- Model: `llama-3.1-13b-instruct` (GGUF, local only)

`app/services/llm_client.rb`:

```ruby
require 'net/http'
require 'json'

class LlmClient
  BASE_URL = "http://localhost:1234/v1/chat/completions"

  def self.generate(prompt, persona: nil)
    uri = URI(BASE_URL)

    messages = []
    if persona
      messages << {
        role: "system",
        content: "You are the following persona:\n\n#{persona.description}"
      }
    end

    messages << { role: "user", content: prompt }

    payload = {
      model: "llama-3.1-13b-instruct",
      messages: messages,
      temperature: 0.7,
      max_tokens: 500
    }

    resp = Net::HTTP.post(uri, payload.to_json, "Content-Type" => "application/json")
    json = JSON.parse(resp.body)
    json.dig("choices", 0, "message", "content")
  end
end
```

This way:

- Persona is injected via the **system message** at inference time.
- Topic description is provided as the **user message**.
- The training datasets still use a text-only instruction built from `full_prompt_for_persona`.

---

### 1.4. Rails Routes (Overview)

`config/routes.rb` roughly looks like:

```ruby
Rails.application.routes.draw do
  resources :personas do
    resources :topics, only: [:new, :create]
  end

  resources :topics do
    member do
      get :rank
      post :submit_rank
    end

    resources :generated_contents, only: [:create]
  end

  root "personas#index"
end
```

You can adjust the root to `topics#index` if you prefer to land on Topics first.

---

### 1.5. Views – High-Level UX

The app uses **simple inline CSS** (no Tailwind/Bootstrap) but provides a clean UI:

- A **navbar** with links:
  - Home / Personas
  - Topics
  - New Topic
- Persona pages:
  - View / create Personas
  - List topics under a Persona
- Topic show page:
  - Shows Persona + Topic prompt
  - Button: **Generate Outputs**
  - Button: **Rank Outputs**
  - List of existing GeneratedContent cards
- Rank page:
  - Displays 3 variants in cards (side-by-side layout)
  - Each card has a dropdown to pick rank **1 (Best), 2 (Middle), 3 (Worst)**

The styling is minimal, intentionally framework-free, and lives in `application.html.erb` using classes like `.container`, `.card`, `.btn`, plus a simple top navbar.

---

## 2. Dataset Exporters (Rails → JSONL)

Exporters live in `rails-app/app/services/` and write JSONL files in the Rails root directory.

### 2.1. SFT Dataset Exporter

```ruby
class SftDatasetExporter
  def self.export(path = "export_sft.jsonl")
    File.open(path, "w") do |file|
      Ranking.find_each do |r|
        best = GeneratedContent.find(r.best_id)

        record = {
          instruction: r.topic.full_prompt_for_persona,
          output: best.content
        }

        file.puts(record.to_json)
      end
    end

    puts "SFT dataset exported to #{path}"
  end
end
```

- Uses **persona-aware instruction** from `Topic#full_prompt_for_persona`
- Uses only the **best** output per ranking

Run from Rails root:

```bash
bin/rails runner "SftDatasetExporter.export"
```

Output: `export_sft.jsonl`

---

### 2.2. DPO Dataset Exporter

```ruby
class RankingDatasetExporter
  def self.export(path = "export_dpo.jsonl")
    File.open(path, "w") do |file|
      Ranking.find_each do |r|
        topic  = r.topic
        best   = GeneratedContent.find(r.best_id)
        middle = GeneratedContent.find(r.middle_id)
        worst  = GeneratedContent.find(r.worst_id)

        instruction = topic.full_prompt_for_persona

        # Best > Middle
        file.puts({
          instruction: instruction,
          chosen: best.content,
          rejected: middle.content
        }.to_json)

        # Best > Worst
        file.puts({
          instruction: instruction,
          chosen: best.content,
          rejected: worst.content
        }.to_json)

        # Middle > Worst
        file.puts({
          instruction: instruction,
          chosen: middle.content,
          rejected: worst.content
        }.to_json)
      end
    end

    puts "DPO dataset exported to #{path}"
  end
end
```

Run:

```bash
bin/rails runner "RankingDatasetExporter.export"
```

Output: `export_dpo.jsonl`

---

### 2.3. Reward Model Dataset Exporter

```ruby
class RmDatasetExporter
  def self.export(path = "export_rm.jsonl")
    File.open(path, "w") do |file|
      Ranking.find_each do |r|
        topic  = r.topic
        best   = GeneratedContent.find(r.best_id)
        middle = GeneratedContent.find(r.middle_id)
        worst  = GeneratedContent.find(r.worst_id)

        instruction = topic.full_prompt_for_persona

        # Best = +1
        file.puts({
          instruction: instruction,
          response: best.content,
          reward: 1
        }.to_json)

        # Middle = 0
        file.puts({
          instruction: instruction,
          response: middle.content,
          reward: 0
        }.to_json)

        # Worst = -1
        file.puts({
          instruction: instruction,
          response: worst.content,
          reward: -1
        }.to_json)
      end
    end

    puts "Reward Model dataset exported to #{path}"
  end
end
```

Run:

```bash
bin/rails runner "RmDatasetExporter.export"
```

Output: `export_rm.jsonl`

---

## 3. Python Training – `model-training/`

All training code lives in `model-training/`. Training **does not run from Rails**.

Training is done on a **Hugging Face base model**, e.g.:

```python
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
```

You must have:

- A Hugging Face account
- Accepted the Meta Llama 3 license
- Logged in via CLI: `hf auth login`

### 3.1. Environment Setup

From `model-training/`:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Typical `requirements.txt` (no bitsandbytes, Mac‑friendly):

```text
transformers>=4.40.0
accelerate>=0.30.0
datasets>=2.18.0
sentencepiece>=0.1.99
tokenizers>=0.19.0
peft>=0.11.1
trl>=0.7.11
typing-extensions>=4.10.0
protobuf>=4.25.3
scipy
numpy
pandas
tqdm>=4.66.0
```

Check MPS:

```bash
python - << 'EOF'
import torch
print("Torch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
EOF
```

You should see `MPS available: True` on Apple Silicon.

---

### 3.2. SFT Training – `train_sft.py`

Key ideas:

- Loads dataset from a JSONL path (CLI option `--dataset`).
- Defaults to `../rails-app/export_sft.jsonl` if not overridden.
- Formats each example as:
  ```
  <s>[INST] {instruction} [/INST]
  {output}</s>
  ```
- Loads base model: `meta-llama/Meta-Llama-3-8B-Instruct`
- Applies LoRA
- Saves to `./sft-llama-3-8b` (or similar folder)

Example command (default path):

```bash
cd model-training
source .venv/bin/activate
python train_sft.py
```

Or with an explicit dataset argument (e.g. a test fixture):

```bash
python train_sft.py --dataset ../rails-app/export_sft_test.jsonl
```

---

### 3.3. DPO Training – `train_dpo.py`

DPO consumes `export_dpo.jsonl` (or another file specified via `--dataset`).

Key ideas:

- Uses columns: `instruction`, `chosen`, `rejected`
- Optionally starts from the SFT checkpoint (`./sft-llama-3-8b`) if it exists; otherwise, starts from `BASE_MODEL`
- Applies LoRA again
- Uses TRL `DPOTrainer`
- Saves to `./dpo-llama-3-8b`

Example commands:

```bash
python train_dpo.py               # uses ../rails-app/export_dpo.jsonl
python train_dpo.py --dataset ../rails-app/export_dpo_test.jsonl
```

---

### 3.4. Reward Model Training – `train_rm.py`

The reward model consumes `export_rm.jsonl` (or a custom path via `--dataset`).

Key ideas:

- Concatenates `instruction + response` → a single `text` field:
  ```
  <s>[INST] {instruction} [/INST]
  {response}</s>
  ```
- Trains a **regression head** (`num_labels = 1`, `problem_type="regression"`) on top of the base model
- Reward values are `+1`, `0`, `-1`
- Saves to `./reward-model-llama-3-8b`

Example:

```bash
python train_rm.py               # uses ../rails-app/export_rm.jsonl
python train_rm.py --dataset ../rails-app/export_rm_test.jsonl
```

This reward model can be used later to score new outputs or as a component in PPO-style RLHF.

---

### 3.5. Merging LoRA to Full HF Model – `merge_lora.py`

Once SFT or DPO is trained, you can merge the LoRA adapters into a standalone Hugging Face model:

```bash
python merge_lora.py \
  --base-model meta-llama/Meta-Llama-3-8B-Instruct \
  --adapter-path ./dpo-llama-3-8b \
  --output-path ./merged-dpo-llama-3-8b
```

Result: a standard HuggingFace model folder at `merged-dpo-llama-3-8b/` containing:

- `config.json`
- `model.safetensors` shards
- `tokenizer.json` etc.

You can do the same for SFT:

```bash
python merge_lora.py \
  --base-model meta-llama/Meta-Llama-3-8B-Instruct \
  --adapter-path ./sft-llama-3-8b \
  --output-path ./merged-sft-llama-3-8b
```

---

### 3.6. Converting Merged Model to GGUF (LM Studio)

To use your fine‑tuned model in **LM Studio**, you convert the merged HF model to GGUF using **llama.cpp**.

Clone llama.cpp once:

```bash
cd ~/dev
git clone https://github.com/ggerganov/llama.cpp.git
```

Convert merged DPO model:

```bash
cd ~/dev/llama.cpp

python convert_hf_to_gguf.py \
  --model ~/dev/youtube-persona-creator/model-training/merged-dpo-llama-3-8b \
  --outfile ~/dev/youtube-persona-creator/gguf-models/llama-3-8b-dpo.q4_k_m.gguf \
  --outtype q4_k_m
```

You can repeat for the SFT model:

```bash
python convert_hf_to_gguf.py \
  --model ~/dev/youtube-persona-creator/model-training/merged-sft-llama-3-8b \
  --outfile ~/dev/youtube-persona-creator/gguf-models/llama-3-8b-sft.q4_k_m.gguf \
  --outtype q4_k_m
```

Then in **LM Studio**, add the generated GGUF file as a local model and run it like any other model.

> Note: The LM Studio “generation model” used during data collection does **not** have to be the same as the HF base model you train. The pipeline is:
>
> - LM Studio (GGUF) → generate + collect rankings
> - HF (Transformers) → train on exported data
> - Convert trained HF model back to GGUF → load in LM Studio

---

## 4. License

MIT License – use at your own risk, no warranty.

You are responsible for complying with the upstream LLaMA 3 license and any additional restrictions on usage or distribution of fine‑tuned models.
