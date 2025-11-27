class Topic < ApplicationRecord
  belongs_to :persona
  has_many :generated_contents, dependent: :destroy

  validates :title, presence: true
  validates :description, presence: true
end
