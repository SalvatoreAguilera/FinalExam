library(shiny)
library(ggplot2)
library(DT)
library(caret)
library(cluster)
library(factoextra)
library(stats)
library(NbClust)
library(mclust)
library(gridExtra)

# Define UI for the application
ui <- fluidPage(
  titlePanel("Advanced Data Analysis and Clustering"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload your CSV File:", accept = ".csv"),
      hr(),
      
      h3("PCA with K-Means Clustering"),
      checkboxGroupInput("pca_kmeans_vars", "Select Variables for PCA and K-Means:", choices = NULL),
      selectInput("species_var", "Select Species Variable (for Validation):", choices = NULL),
      numericInput("num_clusters", "Number of Clusters:", value = 3, min = 2),
      numericInput("num_pca_components", "Number of PCA Components:", value = 2, min = 2, max = 5),
      actionButton("perform_pca_kmeans", "Perform PCA K-Means Clustering"),
      hr(),
      
      h3("Linear Regression"),
      checkboxGroupInput("reg_vars", "Select Predictor Variables:", choices = NULL),
      selectInput("response_var", "Select Response Variable:", choices = NULL),
      numericInput("train_split", "Training Data Split (0-1):", value = 0.7, min = 0.1, max = 0.9, step = 0.05),
      actionButton("build_model", "Build Linear Regression Model"),
      hr(),
      
      h3("Variable Visualization"),
      uiOutput("var_selector")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Data Summary", verbatimTextOutput("summary"), DT::dataTableOutput("data_table")),
        tabPanel("PCA K-Means Clustering", 
                 fluidRow(
                   column(6, 
                          h4("Original Space K-Means"),
                          plotOutput("kmeans_plot"),
                          verbatimTextOutput("kmeans_validation")
                   ),
                   column(6, 
                          h4("PCA K-Means"),
                          plotOutput("pca_kmeans_plot"),
                          verbatimTextOutput("pca_kmeans_validation")
                   )
                 )
        ),
        tabPanel("Linear Regression", 
                 verbatimTextOutput("reg_summary"),
                 tableOutput("model_comparison")),
        tabPanel("Numerical Visualization", plotOutput("numerical_plot")),
        tabPanel("Categorical Visualization", plotOutput("categorical_plot"))
      )
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  
  # Reactive value to store the dataset
  data <- reactiveVal()
  
  # Load dataset and update variable choices
  observeEvent(input$file, {
    req(input$file)
    dataset <- read.csv(input$file$datapath)
    
    # Convert character columns to factors
    dataset[] <- lapply(dataset, function(x) {
      if(is.character(x)) as.factor(x) else x
    })
    
    num_vars <- names(dataset)[sapply(dataset, is.numeric)]
    cat_vars <- names(dataset)[sapply(dataset, is.factor)]
    
    data(dataset)
    
    updateCheckboxGroupInput(session, "pca_kmeans_vars", choices = num_vars)
    updateCheckboxGroupInput(session, "reg_vars", choices = num_vars)
    updateSelectInput(session, "response_var", choices = num_vars)
    updateSelectInput(session, "species_var", choices = cat_vars)
    
    # Dynamically create variable selector for visualization
    output$var_selector <- renderUI({
      tagList(
        h4("Numerical Variables"),
        checkboxGroupInput("num_var_choices", "Select Numerical Variable(s):", choices = num_vars),
        h4("Categorical Variables"),
        checkboxGroupInput("cat_var_choices", "Select Categorical Variable(s):", choices = cat_vars),
        actionButton("generate_plots", "Generate Plots")
      )
    })
  })
  
  # Display dataset summary and table
  output$summary <- renderPrint({
    req(data())
    summary(data())
  })
  
  output$data_table <- DT::renderDataTable({
    req(data())
    DT::datatable(data())
  })
  
  # Integrated PCA and K-Means Clustering
  observeEvent(input$perform_pca_kmeans, {
    req(data(), input$pca_kmeans_vars, input$num_clusters, input$species_var)
    
    # Prepare data for clustering
    cluster_data <- data()[, input$pca_kmeans_vars]
    species_data <- data()[[input$species_var]]
    
    # Perform scaling
    scaled_data <- scale(cluster_data)
    
    # Perform K-Means on original scaled data
    set.seed(123)
    kmeans_result <- kmeans(scaled_data, centers = input$num_clusters)
    
    # Visualization for original space K-Means
    output$kmeans_plot <- renderPlot({
      # Use first two variables for 2D plot
      plot(scaled_data[,1], scaled_data[,2], 
           col = kmeans_result$cluster, 
           pch = 19, 
           main = "K-Means Clustering (Original Space)")
      points(kmeans_result$centers[,1:2], col = 1:input$num_clusters, 
             pch = 8, cex = 2)
    })
    
    # K-Means Clustering Validation
    output$kmeans_validation <- renderPrint({
      # Adjusted Rand Index for comparing clustering to actual species
      cluster_labels <- kmeans_result$cluster
      rand_index <- adjustedRandIndex(cluster_labels, species_data)
      
      # Compute cluster purity
      purity <- function(clusters, true_labels) {
        # Create a contingency table
        cont_table <- table(true_labels, clusters)
        
        # Find the maximum count for each cluster
        cluster_max <- apply(cont_table, 2, max)
        
        # Sum of max counts divided by total number of observations
        sum(cluster_max) / length(true_labels)
      }
      
      cluster_purity <- purity(cluster_labels, species_data)
      
      cat("K-Means Clustering Validation (Original Space):\n")
      cat("Number of Clusters:", input$num_clusters, "\n")
      cat("Adjusted Rand Index:", rand_index, "\n")
      cat("Cluster Purity:", cluster_purity, "\n")
      cat("\nCluster Distribution:\n")
      print(table(cluster_labels, species_data))
    })
    
    # Perform PCA
    pca_result <- prcomp(scaled_data, scale. = TRUE)
    
    # Select specified number of components
    pca_components <- pca_result$x[, 1:input$num_pca_components]
    
    # Perform K-Means on PCA components
    set.seed(123)
    kmeans_pca_result <- kmeans(pca_components, centers = input$num_clusters)
    
    # PCA K-Means Visualization
    output$pca_kmeans_plot <- renderPlot({
      # Use first two PCA components for visualization
      plot(pca_components[,1], pca_components[,2], 
           col = kmeans_pca_result$cluster, 
           pch = 19, 
           main = paste("PCA K-Means Clustering (", input$num_pca_components, " Components)"))
      points(kmeans_pca_result$centers[,1:2], col = 1:input$num_clusters, 
             pch = 8, cex = 2)
    })
    
    # PCA K-Means Validation
    output$pca_kmeans_validation <- renderPrint({
      # Adjusted Rand Index for comparing clustering to actual species
      cluster_labels <- kmeans_pca_result$cluster
      rand_index <- adjustedRandIndex(cluster_labels, species_data)
      
      # Compute cluster purity
      purity <- function(clusters, true_labels) {
        # Create a contingency table
        cont_table <- table(true_labels, clusters)
        
        # Find the maximum count for each cluster
        cluster_max <- apply(cont_table, 2, max)
        
        # Sum of max counts divided by total number of observations
        sum(cluster_max) / length(true_labels)
      }
      
      cluster_purity <- purity(cluster_labels, species_data)
      
      cat("PCA K-Means Clustering Validation:\n")
      cat("Number of PCA Components:", input$num_pca_components, "\n")
      cat("Number of Clusters:", input$num_clusters, "\n")
      cat("Adjusted Rand Index:", rand_index, "\n")
      cat("Cluster Purity:", cluster_purity, "\n")
      cat("\nCluster Distribution:\n")
      print(table(cluster_labels, species_data))
    })
  })
  
  # Linear Regression
  observeEvent(input$build_model, {
    req(data(), input$reg_vars, input$response_var, input$train_split)
    
    # Prepare data for regression
    model_data <- data()[, c(input$reg_vars, input$response_var)]
    
    # Split the data into training and testing sets
    set.seed(123)
    train_index <- createDataPartition(model_data[[input$response_var]], 
                                       p = input$train_split, 
                                       list = FALSE)
    train_data <- model_data[train_index, ]
    test_data <- model_data[-train_index, ]
    
    # Create formula
    formula <- as.formula(paste(input$response_var, "~", 
                                paste(input$reg_vars, collapse = "+")))
    
    # Fit linear regression model
    model <- lm(formula, data = train_data)
    
    # Predictions on test data
    predictions <- predict(model, test_data)
    
    # Calculate metrics
    mse <- mean((test_data[[input$response_var]] - predictions)^2)
    rmse <- sqrt(mse)
    r_squared <- summary(model)$r.squared
    adj_r_squared <- summary(model)$adj.r.squared
    
    # Display model summary
    output$reg_summary <- renderPrint({
      summary(model)
    })
    
    # Display model comparison metrics
    output$model_comparison <- renderTable({
      data.frame(
        Metric = c("Mean Squared Error", "Root Mean Squared Error", 
                   "R-squared", "Adjusted R-squared"),
        Value = c(mse, rmse, r_squared, adj_r_squared)
      )
    })
  })
  
  # Generate Numerical and Categorical Visualizations
  observeEvent(input$generate_plots, {
    req(data())
    
    # Numerical Visualization
    output$numerical_plot <- renderPlot({
      req(input$num_var_choices)
      
      # Create plots for numerical variables
      plots <- lapply(input$num_var_choices, function(var) {
        ggplot(data(), aes_string(x = var)) +
          geom_histogram(bins = 30, fill = "blue", color = "black") +
          labs(title = paste("Histogram of", var), x = var, y = "Frequency") +
          theme_minimal()
      })
      
      # Use gridExtra to arrange multiple plots
      if (length(plots) > 1) {
        gridExtra::grid.arrange(grobs = plots, ncol = 2)
      } else if (length(plots) == 1) {
        plots[[1]]
      }
    })
    
    # Categorical Visualization
    output$categorical_plot <- renderPlot({
      req(input$cat_var_choices)
      
      # Create bar charts for categorical variables
      plots <- lapply(input$cat_var_choices, function(var) {
        ggplot(data(), aes_string(x = var)) +
          geom_bar(fill = "green", color = "black") +
          labs(title = paste("Bar Chart of", var), x = var, y = "Count") +
          theme_minimal() +
          theme(axis.text.x = element_text(angle = 45, hjust = 1))
      })
      
      # Use gridExtra to arrange multiple plots
      if (length(plots) > 1) {
        gridExtra::grid.arrange(grobs = plots, ncol = 2)
      } else if (length(plots) == 1) {
        plots[[1]]
      }
    })
  })
}

# Run the application
shinyApp(ui = ui, server = server)