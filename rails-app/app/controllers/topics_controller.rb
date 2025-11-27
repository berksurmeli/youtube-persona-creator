class TopicsController < ApplicationController
  def index
    @topics   = Topic.includes(:persona).order(created_at: :desc)
    @personas = Persona.all
  end

  def show
    @topic    = Topic.find(params[:id])
    @persona  = @topic.persona
    @contents = @topic.generated_contents.order(:variant_index)
  end

  def new
    @topic    = Topic.new
    @personas = Persona.all
  end

  def create
    @topic = Topic.new(topic_params)

    if @topic.save
      redirect_to @topic, notice: "Topic created!"
    else
      @personas = Persona.all
      render :new, status: :unprocessable_entity
    end
  end

  def rank
    @topic    = Topic.find(params[:id])
    @persona  = @topic.persona
    @contents = @topic.generated_contents.order(:variant_index)
  end

  def submit_rank
    topic = Topic.find(params[:id])
    ranks = params[:ranking] || {}  # { "content_id" => "1" / "2" / "3" }

    content_ids_by_rank = ranks.sort_by { |_, rank| rank.to_i }
    best_id, middle_id, worst_id = content_ids_by_rank.map(&:first)

    Ranking.create!(
      topic:      topic,
      best_id:    best_id,
      middle_id:  middle_id,
      worst_id:   worst_id
    )

    redirect_to topic_path(topic), notice: "Ranking saved!"
  end

  private

  def topic_params
    params.require(:topic).permit(:title, :description, :persona_id)
  end
end
