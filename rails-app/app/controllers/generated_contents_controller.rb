class GeneratedContentsController < ApplicationController
  VARIANTS = 3

  def create
    topic = Topic.find(params[:topic_id])

    VARIANTS.times do |i|
      llm_output = LlmClient.generate(
        topic.full_prompt_for_persona,
        persona: topic.persona
      )


      topic.generated_contents.create!(
        content: llm_output,
        variant_index: i
      )
    end

    redirect_to rank_topic_path(topic)
  end
end
