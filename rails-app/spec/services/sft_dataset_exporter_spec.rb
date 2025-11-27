require "rails_helper"

RSpec.describe SftDatasetExporter do
  let(:persona) { Persona.create!(name: "Test Persona", description: "Persona description.") }
  let(:topic) { Topic.create!(persona: persona, title: "T1", description: "Test topic prompt") }

  let!(:best)   { topic.generated_contents.create!(content: "Best content", variant_index: 0) }
  let!(:middle) { topic.generated_contents.create!(content: "Middle content", variant_index: 1) }
  let!(:worst)  { topic.generated_contents.create!(content: "Worst content", variant_index: 2) }

  let!(:ranking) do
    Ranking.create!(
      topic: topic,
      best_id: best.id,
      middle_id: middle.id,
      worst_id: worst.id
    )
  end

  let(:path) { Rails.root.join("tmp/sft_test.jsonl") }

  after { File.delete(path) if File.exist?(path) }

  it "exports correct SFT entries" do
    SftDatasetExporter.export(path)

    lines = File.read(path).split("\n")
    expect(lines.size).to eq(1)

    json = JSON.parse(lines.first)

    expect(json["instruction"]).to include(persona.description)
    expect(json["output"]).to eq("Best content")
  end
end
