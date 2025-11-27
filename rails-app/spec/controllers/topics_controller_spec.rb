require "rails_helper"

RSpec.describe TopicsController, type: :controller do
  describe "GET #index" do
    it "returns success" do
      get :index
      expect(response).to have_http_status(:success)
    end
  end

  describe "POST #create" do
    it "creates a new topic" do
      persona = create(:persona)

      expect {
        post :create, params: {
          topic: { persona_id: persona.id, title: "T1", description: "D1" }
        }
      }.to change(Topic, :count).by(1)
    end
  end
end
