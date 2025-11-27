class Topic < ApplicationRecord
  belongs_to :persona
  has_many :generated_contents, dependent: :destroy

  validates :title, presence: true
  validates :description, presence: true

  def full_prompt_for_persona
    <<~PROMPT
    You are the following persona:

    #{persona.name}
    #{persona.description}

    Now respond to the following request:

    #{description}
    PROMPT
  end
end
