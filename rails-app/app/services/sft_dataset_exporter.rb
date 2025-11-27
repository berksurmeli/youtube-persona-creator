class SftDatasetExporter < ApplicationExporter
  def self.export(path = "export_sft.jsonl")
    File.open(path, "w") do |file|
      Ranking.find_each do |r|
        topic = r.topic
        winner = GeneratedContent.find(r.best_id)

        record = {
          instruction: build_instruction(topic),
          output: winner.content
        }

        file.puts(record.to_json)
      end
    end

    puts "SFT dataset exported to #{path}"
  end
end
