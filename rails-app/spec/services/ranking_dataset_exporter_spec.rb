require "rails_helper"

RSpec.describe RankingDatasetExporter do
  let(:persona) { Persona.create!(name: "Persona", description: "Persona desc") }
  let(:topic)   { Topic.create!(persona: persona, title: "T1", description: "Topic prompt") }

  let!(:best)   { topic.generated_contents.create!(content: "Best", variant_index: 0) }
  let!(:middle) { topic.generated_contents.create!(content: "Middle", variant_index: 1) }
  let!(:worst)  { topic.generated_contents.create!(content: "Worst", variant_index: 2) }

  let!(:ranking) do
    Ranking.create!(
      topic: topic,
      best_id: best.id,
      middle_id: middle.id,
      worst_id: worst.id
    )
  end

  let(:path) { Rails.root.join("tmp/dpo_test.jsonl") }

  after { File.delete(path) if File.exist?(path) }

  it "exports 3 DPO preference pairs" do
    RankingDatasetExporter.export(path)

    lines = File.read(path).split("\n")
    expect(lines.size).to eq(3)

    entries = lines.map { |l| JSON.parse(l) }

    # Best > Middle
    expect(entries[0]["chosen"]).to eq("Best")
    expect(entries[0]["rejected"]).to eq("Middle")

    # Best > Worst
    expect(entries[1]["chosen"]).to eq("Best")
    expect(entries[1]["rejected"]).to eq("Worst")

    # Middle > Worst
    expect(entries[2]["chosen"]).to eq("Middle")
    expect(entries[2]["rejected"]).to eq("Worst")

    entries.each do |json|
      expect(json["instruction"]).to include(persona.description)
    end
  end
end
