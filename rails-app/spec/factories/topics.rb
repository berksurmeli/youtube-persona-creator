FactoryBot.define do
  factory :topic do
    association :persona
    title { "Test Topic" }
    description { "Explain the importance of testing." }
  end
end
