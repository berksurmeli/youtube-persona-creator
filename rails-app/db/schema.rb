# This file is auto-generated from the current state of the database. Instead
# of editing this file, please use the migrations feature of Active Record to
# incrementally modify your database, and then regenerate this schema definition.
#
# This file is the source Rails uses to define your schema when running `bin/rails
# db:schema:load`. When creating a new database, `bin/rails db:schema:load` tends to
# be faster and is potentially less error prone than running all of your
# migrations from scratch. Old migrations may fail to apply correctly if those
# migrations use external dependencies or application code.
#
# It's strongly recommended that you check this file into your version control system.

ActiveRecord::Schema[8.1].define(version: 2025_11_27_132216) do
  # These are extensions that must be enabled in order to support this database
  enable_extension "pg_catalog.plpgsql"

  create_table "feedbacks", force: :cascade do |t|
    t.string "category"
    t.text "comment"
    t.datetime "created_at", null: false
    t.bigint "generated_content_id", null: false
    t.integer "rating"
    t.datetime "updated_at", null: false
    t.index ["generated_content_id"], name: "index_feedbacks_on_generated_content_id"
  end

  create_table "generated_contents", force: :cascade do |t|
    t.text "content"
    t.datetime "created_at", null: false
    t.bigint "topic_id", null: false
    t.datetime "updated_at", null: false
    t.integer "variant_index", default: 0
    t.index ["topic_id"], name: "index_generated_contents_on_topic_id"
  end

  create_table "personas", force: :cascade do |t|
    t.datetime "created_at", null: false
    t.text "description"
    t.string "name"
    t.datetime "updated_at", null: false
  end

  create_table "rankings", force: :cascade do |t|
    t.integer "best_id"
    t.datetime "created_at", null: false
    t.integer "loser_id"
    t.integer "middle_id"
    t.integer "tie"
    t.bigint "topic_id", null: false
    t.datetime "updated_at", null: false
    t.integer "winner_id"
    t.integer "worst_id"
    t.index ["topic_id"], name: "index_rankings_on_topic_id"
  end

  create_table "topics", force: :cascade do |t|
    t.datetime "created_at", null: false
    t.text "description"
    t.bigint "persona_id", null: false
    t.string "title"
    t.datetime "updated_at", null: false
    t.index ["persona_id"], name: "index_topics_on_persona_id"
  end

  add_foreign_key "feedbacks", "generated_contents"
  add_foreign_key "generated_contents", "topics"
  add_foreign_key "rankings", "topics"
  add_foreign_key "topics", "personas"
end
