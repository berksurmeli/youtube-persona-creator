require "rails_helper"

RSpec.describe Topic, type: :model do
  it "is valid with required attributes" do
    expect(build(:topic)).to be_valid
  end

  it "belongs to a persona" do
    expect(build(:topic).persona).to be_present
  end

  it "returns a full prompt with persona context" do
    topic = build(:topic)
    expect(topic.full_prompt_for_persona).to include(topic.persona.description)
  end
end
