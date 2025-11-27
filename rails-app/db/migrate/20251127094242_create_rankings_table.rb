class CreateRankingsTable < ActiveRecord::Migration[8.1]
  def change
   create_table :rankings do |t|
      t.references :topic, null: false, foreign_key: true
      t.integer :winner_id
      t.integer :loser_id
      t.integer :tie
      t.timestamps
    end
  end
end
