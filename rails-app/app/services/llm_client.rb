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
