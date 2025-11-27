namespace :pipeline do
  desc "Full end-to-end RLHF pipeline test with dummy data"
  task test: :environment do

    puts "=== Creating Dummy Persona ==="
    persona = Persona.create!(
      name: "Test Persona",
      description: "A temporary persona used only for pipeline testing."
    )

    puts "=== Creating Dummy Topic ==="
    topic = persona.topics.create!(
      title: "Test Topic",
      description: "Write a short test response about testing pipelines."
    )

    puts "=== Creating Dummy Outputs ==="
    # Instead of calling LM Studio, we stub content:
    contents = []
    contents << topic.generated_contents.create!(content: "Best response example",   variant_index: 0)
    contents << topic.generated_contents.create!(content: "Middle response example", variant_index: 1)
    contents << topic.generated_contents.create!(content: "Worst response example",  variant_index: 2)

    puts "=== Creating Dummy Ranking ==="
    Ranking.create!(
      topic: topic,
      best_id: contents[0].id,
      middle_id: contents[1].id,
      worst_id: contents[2].id
    )

    puts "=== Running Exporters ==="
    SftDatasetExporter.export("export_sft_test.jsonl")
    RankingDatasetExporter.export("export_dpo_test.jsonl")
    RmDatasetExporter.export("export_rm_test.jsonl")

    puts "=== Export Complete ==="

    puts "=== CLEANUP INSTRUCTIONS ==="
    puts ""
    puts "After running the python scripts on the test jsonl files, run:"
    puts ""
    puts "rails runner 'Persona.where(name: \"Test Persona\").destroy_all'"
    puts "rm rails-app/export_*_test.jsonl"
    puts ""
    puts "=== END ==="
  end
end
