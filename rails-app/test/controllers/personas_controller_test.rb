require "test_helper"

class PersonasControllerTest < ActionDispatch::IntegrationTest
  test "should get index" do
    get personas_index_url
    assert_response :success
  end

  test "should get show" do
    get personas_show_url
    assert_response :success
  end

  test "should get new" do
    get personas_new_url
    assert_response :success
  end

  test "should get create" do
    get personas_create_url
    assert_response :success
  end
end
