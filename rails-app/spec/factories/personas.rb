FactoryBot.define do
  factory :persona do
    name { Faker::Name.unique.name }
    description { "This persona speaks in a distinctive style for testing." }
  end
end
