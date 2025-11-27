class CreateGeneratedContentsTable < ActiveRecord::Migration[8.1]
  def change
    create_table :generated_contents do |t|
      t.references :topic, null: false, foreign_key: true
      t.text :content
      t.integer :variant_index, default: 0
      t.timestamps
    end
  end
end
