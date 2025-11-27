require "rails_helper"

RSpec.describe Persona, type: :model do
  it "is valid with valid attributes" do
    expect(build(:persona)).to be_valid
  end

  it "requires a name" do
    expect(build(:persona, name: nil)).not_to be_valid
  end

  it "requires a description" do
    expect(build(:persona, description: nil)).not_to be_valid
  end
end
