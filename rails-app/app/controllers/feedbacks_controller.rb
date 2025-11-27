class FeedbacksController < ApplicationController
  def create
    @content = GeneratedContent.find(params[:generated_content_id])

    @content.feedbacks.create!(
      rating: params[:rating],
      category: params[:category],
      comment: params[:comment]
    )

    redirect_to topic_path(@content.topic), notice: "Feedback saved!"
  end
end
