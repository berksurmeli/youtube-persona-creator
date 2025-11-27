class CreatePersonas < ActiveRecord::Migration[8.1]
  def change
    create_table :personas do |t|
      t.string :name
      t.text :description

      t.timestamps
    end
  end
end
