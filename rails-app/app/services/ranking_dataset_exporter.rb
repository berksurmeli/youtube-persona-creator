class RankingDatasetExporter < ApplicationExporter
  def self.export(path = "export_dpo.jsonl")
    File.open(path, "w") do |file|
      Ranking.find_each do |r|
        topic = r.topic

        best   = GeneratedContent.find(r.best_id)
        middle = GeneratedContent.find(r.middle_id)
        worst  = GeneratedContent.find(r.worst_id)

        instruction = build_instruction(topic)

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
