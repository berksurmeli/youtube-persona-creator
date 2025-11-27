require "rails_helper"

RSpec.describe RmDatasetExporter do
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

  let(:path) { Rails.root.join("tmp/rm_test.jsonl") }

  after { File.delete(path) if File.exist?(path) }

  it "exports 3 reward entries with correct reward values" do
    RmDatasetExporter.export(path)

    lines = File.read(path).split("\n")
    expect(lines.size).to eq(3)

    entries = lines.map { |l| JSON.parse(l) }

    expect(entries[0]["response"]).to eq("Best")
    expect(entries[0]["reward"]).to eq(1)

    expect(entries[1]["response"]).to eq("Middle")
    expect(entries[1]["reward"]).to eq(0)

    expect(entries[2]["response"]).to eq("Worst")
    expect(entries[2]["reward"]).to eq(-1)

    entries.each do |json|
      expect(json["instruction"]).to include(persona.description)
    end
  end
end
