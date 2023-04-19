# We strongly recommend using the required_providers block to set the
# Azure Provider source and version being used

#References: https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/resources/app_service_source_control

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "=3.0.0"
    }
  }
}

# Configure the Microsoft Azure Provider
provider "azurerm" {
  features {}
}

# Create a resource group
resource "azurerm_resource_group" "example" {
  name     = "fuel50_test_rg"
  location = "eastus"
}

# Create the Linux App Service Plan
resource "azurerm_service_plan" "appserviceplan" {
  name                = "Fuel50_rg"
  location            = azurerm_resource_group.example.location
  resource_group_name = azurerm_resource_group.example.name
  os_type             = "Linux"
  sku_name            = "F1"
}

# Create the web app, pass in the App Service Plan ID
resource "azurerm_linux_web_app" "webapp" {
  name                  = "Fuel50churnpredict"
  location              = azurerm_resource_group.example.location
  resource_group_name   = azurerm_resource_group.example.name
  service_plan_id       = azurerm_service_plan.appserviceplan.id
  https_only            = true
  site_config { 
    minimum_tls_version = "1.2"
  }
}

#  Deploy code from a public docker repo
resource "azurerm_app_service_source_control" "sourcecontrol" {
  app_id             = azurerm_linux_web_app.webapp.id
  github_action_configuration {
     container_configuration {
       image_name 	   = "fuel50churnpredict"
       registry_url       = "anderhong/fuel50churnpredict:1.0"
     }  
  }
}


