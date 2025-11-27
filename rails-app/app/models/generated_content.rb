class GeneratedContent < ApplicationRecord
  belongs_to :topic
  has_many :feedbacks, dependent: :destroy
end
