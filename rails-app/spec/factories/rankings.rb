FactoryBot.define do
  factory :ranking do
    association :topic

    after(:build) do |ranking|
      topic = ranking.topic

      gc1 = create(:generated_content, topic: topic, variant_index: 0)
      gc2 = create(:generated_content, topic: topic, variant_index: 1)
      gc3 = create(:generated_content, topic: topic, variant_index: 2)

      ranking.best_id   = gc1.id
      ranking.middle_id = gc2.id
      ranking.worst_id  = gc3.id
    end
  end
end
