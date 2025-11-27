class Persona < ApplicationRecord
  has_many :topics, dependent: :destroy

  validates :name, presence: true
  validates :description, presence: true
end
