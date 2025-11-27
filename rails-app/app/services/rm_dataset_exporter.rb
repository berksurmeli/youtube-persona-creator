class RmDatasetExporter < ApplicationExporter
  def self.export(path = "export_rm.jsonl")
    File.open(path, "w") do |file|
      Ranking.find_each do |r|
        topic = r.topic

        best   = GeneratedContent.find(r.best_id)
        middle = GeneratedContent.find(r.middle_id)
        worst  = GeneratedContent.find(r.worst_id)

        instruction = build_instruction(topic)

        file.puts({
          instruction: instruction,
          response: best.content,
          reward: 1
        }.to_json)

        file.puts({
          instruction: instruction,
          response: middle.content,
          reward: 0
        }.to_json)

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
