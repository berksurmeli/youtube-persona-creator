class ApplicationExporter
  def self.build_instruction(topic)
    persona = topic.persona

    [
      "[persona: #{persona.name}]",
      "[persona_description: #{persona.description}]",
      "[task: #{topic.description}]"
    ].join("\n")
  end
end