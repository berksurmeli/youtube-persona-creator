class PersonasController < ApplicationController
  def index
    @personas = Persona.all
  end

  def show
    @persona = Persona.find(params[:id])
    @topics  = @persona.topics.order(created_at: :desc)
  end

  def new
    @persona = Persona.new
  end

  def create
    @persona = Persona.new(persona_params)

    if @persona.save
      redirect_to @persona, notice: "Persona created!"
    else
      render :new, status: :unprocessable_entity
    end
  end

  private

  def persona_params
    params.require(:persona).permit(:name, :description)
  end
end
