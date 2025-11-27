Rails.application.routes.draw do
  root "topics#index"

  resources :topics do
    resources :generated_contents, only: [:create]

    member do
      get  :rank
      post :submit_rank
    end
  end

  resources :personas do
    resources :topics, only: [:new, :create, :index]
  end
end
