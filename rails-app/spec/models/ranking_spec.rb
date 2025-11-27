require "rails_helper"

RSpec.describe Ranking, type: :model do
  it "creates ranking with 3 relationships" do
    ranking = create(:ranking)

    expect(ranking.best).to be_present
    expect(ranking.middle).to be_present
    expect(ranking.worst).to be_present
  end
end
