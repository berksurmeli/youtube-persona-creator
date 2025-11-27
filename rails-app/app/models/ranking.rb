class Ranking < ApplicationRecord
  belongs_to :topic

  belongs_to :best, class_name: "GeneratedContent"
  belongs_to :middle, class_name: "GeneratedContent"
  belongs_to :worst, class_name: "GeneratedContent"
end
