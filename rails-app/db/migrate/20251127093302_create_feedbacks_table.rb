class CreateFeedbacksTable < ActiveRecord::Migration[8.1]
  def change
    create_table :feedbacks do |t|
      t.references :generated_content, null: false, foreign_key: true
      t.integer :rating  # 1 = bad, 2 = neutral, 3 = good
      t.string :category # optional: "tone", "too_generic", etc
      t.text :comment
      t.timestamps
    end
  end
end
