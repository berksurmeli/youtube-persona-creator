# YouTube Persona Creator – RLHF Training Stack

This repository contains a complete local workflow for **collecting RLHF-style feedback** and **fine‑tuning a local LLM (Llama 3.1 13B Instruct)** to generate content in different *personas*.

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
      result = LlmClient.generate(topic.full_prompt_for_persona)
      topic.generated_contents.create!(
        content: result,
        variant_index: i
      )
    end

    redirect_to topic_path(topic), notice: "Generated #{VARIANTS} outputs!"
  end
end
```

> `full_prompt_for_persona` is typically a helper on `Topic` that combines Persona + Topic, e.g.:
>
> ```ruby
> def full_prompt_for_persona
>   <<~PROMPT
>   You are the following persona:
>   #{persona.name} - #{persona.description}
>
>   Generate content for this request:
>   #{description}
>   PROMPT
> end
> ```

This ensures **Persona context is embedded into every instruction** and will therefore influence training.

---

### 1.3. LLM Client (LM Studio Integration)

LM Studio runs a local server (OpenAI-compatible) e.g. at:

- `http://localhost:1234/v1/chat/completions`  
- Model: `llama-3.1-13b-instruct`

`app/services/llm_client.rb`:

```ruby
require 'net/http'
require 'json'

class LlmClient
  BASE_URL = "http://localhost:1234/v1/chat/completions"

  def self.generate(prompt)
    uri = URI(BASE_URL)

    payload = {
      model: "llama-3.1-13b-instruct",
      messages: [
        { role: "user", content: prompt }
      ],
      temperature: 0.7,
      max_tokens: 500
    }

    resp = Net::HTTP.post(uri, payload.to_json, "Content-Type" => "application/json")
    json = JSON.parse(resp.body)
    json["choices"][0]["message"]["content"]
  end
end
```

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
  - Displays 3 variants in cards
  - Each card has a dropdown to pick rank **1 (Best), 2 (Middle), 3 (Worst)**

The styling is minimal, intentionally framework-free, and lives in `application.html.erb` using a `.container`, `.card`, `.btn`, etc.

---

## 2. Dataset Exporters (Rails → JSONL)

Exporters live in `rails-app/app/services/` and write JSONL files in the Rails root directory.

### 2.1. SFT Dataset Exporter

```ruby
class SftDatasetExporter
  def self.export(path = "export_sft.jsonl")
    File.open(path, "w") do |file|
      Ranking.find_each do |r|
        winner = GeneratedContent.find(r.best_id)

        record = {
          instruction: r.topic.full_prompt_for_persona,
          output: winner.content
        }

        file.puts(record.to_json)
      end
    end

    puts "SFT dataset exported to #{path}"
  end
end
```

- Uses **persona-aware instruction** from `Topic#full_prompt_for_persona`
- Uses only the **best** output

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

---

### 3.2. SFT Training – `train_sft.py`

Key ideas:

- Loads `../rails-app/export_sft.jsonl`
- Formats data as `<s>[INST] ... [/INST]\n output </s>`
- Loads base model: `meta-llama/Meta-Llama-3.1-13B-Instruct`
- Applies LoRA
- Saves `./sft-llama-3.1-13b`

Run:

```bash
cd model-training
source .venv/bin/activate
python train_sft.py
```

---

### 3.3. DPO Training – `train_dpo.py`

Key ideas:

- Loads `../rails-app/export_dpo.jsonl`
- Uses columns: `instruction`, `chosen`, `rejected`
- Optionally starts from SFT checkpoint (`./sft-llama-3.1-13b`)
- Applies LoRA again
- Uses TRL `DPOTrainer`
- Saves `./dpo-llama-3.1-13b`

Run:

```bash
python train_dpo.py
```

---

### 3.4. Reward Model Training – `train_rm.py`

Key ideas:

- Loads `../rails-app/export_rm.jsonl`
- Uses `instruction + response` → text
- Trains a **regression head** on top of LLaMA (`num_labels = 1`, `problem_type="regression"`)
- Reward values: `+1`, `0`, `-1`
- Saves `./reward-model-llama-3.1-13b`

Run:

```bash
python train_rm.py
```

This can later be used to score new model outputs or drive PPO-style RLHF.

---

### 3.5. Merging LoRA to Full HF Model – `merge_lora.py`

Once SFT/DPO are trained, you can merge adapters into a standalone HF model.

Example usage:

```bash
python merge_lora.py \
  --base-model meta-llama/Meta-Llama-3.1-13B-Instruct \
  --adapter-path ./dpo-llama-3.1-13b \
  --output-path ./merged-dpo-llama-3.1-13b
```

Result: a standard HuggingFace model folder at `merged-dpo-llama-3.1-13b/`.

---

### 3.6. Converting Merged Model to GGUF (LM Studio)

Using `llama.cpp`:

```bash
cd ~/dev
git clone https://github.com/ggerganov/llama.cpp.git

cd llama.cpp

python convert_hf_to_gguf.py \
  --model ~/dev/youtube-persona-creator/model-training/merged-dpo-llama-3.1-13b \
  --outfile ~/dev/youtube-persona-creator/gguf-models/llama-3.1-13b-dpo.q4_k_m.gguf \
  --outtype q4_k_m
```

You can repeat for SFT:

```bash
python convert_hf_to_gguf.py \
  --model ~/dev/youtube-persona-creator/model-training/merged-sft-llama-3.1-13b \
  --outfile ~/dev/youtube-persona-creator/gguf-models/llama-3.1-13b-sft.q4_k_m.gguf \
  --outtype q4_k_m
```

Then in **LM Studio**, add the generated GGUF file as a local model.

---

## 4. How Personas Influence Training

Personas are **not just UI metadata** – they are **baked into the training data** by:

1. Combining persona name + description with the topic prompt inside `Topic#full_prompt_for_persona`.
2. Using that **combined text** as the `instruction` in all exported datasets (SFT / DPO / RM).

That means:

- During SFT, the model learns:  
  “When I see *this persona context + this topic*, the best style of output is *this*.”
- During DPO, preference learning learns which outputs better match the persona’s style.
- During reward modeling, the reward head learns to give higher scores to outputs that match the persona’s intent.

Over time, as you collect more data for each Persona, you build a **persona-conditioned LLM** that can generate content aligned with different YouTube “characters.”

---

## 5. Typical Workflow

1. **Create Persona** (e.g. “Short-form aggressive finance coach”).
2. **Create Topic** under that Persona (video idea/prompt).
3. Click **Generate Outputs** → Rails calls LM Studio → stores 3 variants.
4. Click **Rank Outputs**:
   - Assign 1 / 2 / 3 as Best / Middle / Worst.
5. Repeat for many topics / personas.
6. Periodically run exporters in `rails-app/`:
   - `SftDatasetExporter.export`
   - `RankingDatasetExporter.export`
   - `RmDatasetExporter.export`
7. In `model-training/`:
   - Train SFT → `train_sft.py`
   - Train DPO → `train_dpo.py`
   - Optionally train RM → `train_rm.py`
   - Merge adapters → `merge_lora.py`
   - Convert to GGUF → `llama.cpp` script
8. Load GGUF model into **LM Studio** and use your **persona-tuned** Llama model locally.

---

## 6. Notes & Future Improvements

- Add **dataset stats page** (count of Persons, Topics, Generations, Rankings).
- Add **export buttons** in Rails UI to trigger exporters from the browser (calling `system` or background jobs).
- Add **more structured feedback** (tags like “too generic”, “wrong tone”, etc.).
- Add **automatic eval** using the reward model against new generations.

---

## 7. License

MIT License – use at your own risk, no warranty.

