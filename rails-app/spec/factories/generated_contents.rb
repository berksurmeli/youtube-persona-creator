FactoryBot.define do
  factory :generated_content do
    association :topic
    content { "This is a generated test output." }
    variant_index { 0 }
  end
end
