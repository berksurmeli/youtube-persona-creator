class AddRanksToRankings < ActiveRecord::Migration[8.1]
  def change
    add_column :rankings, :best_id, :integer
    add_column :rankings, :middle_id, :integer
    add_column :rankings, :worst_id, :integer
  end
end
