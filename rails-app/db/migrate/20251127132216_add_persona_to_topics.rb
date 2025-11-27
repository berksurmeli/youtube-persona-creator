class AddPersonaToTopics < ActiveRecord::Migration[8.1]
  def change
    add_reference :topics, :persona, null: false, foreign_key: true
  end
end
