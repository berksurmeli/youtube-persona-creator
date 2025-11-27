require "rails_helper"

RSpec.describe GeneratedContent, type: :model do
  it "is valid" do
    expect(build(:generated_content)).to be_valid
  end
end
