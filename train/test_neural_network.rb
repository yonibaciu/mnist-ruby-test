#!/usr/bin/env ruby

require 'rubygems'
require 'bundler/setup'
require 'ruby-fann' # gem install ruby-fann
require 'pry'
require './mnist_loader'
require './cropper'

class TestNeuralNetwork

  TEST_SIZE = 10000

  def self.run(fann)
    puts "\nLoading test data..."
    inputs, expected_outputs = MnistLoader.test_set.get_data_and_labels(TEST_SIZE)
    
    puts "Testing the trained network with #{inputs.length} examples..."
    
    success, failure, errsum = 0,0,0
    inputs.each.with_index do |input, i|
      output = fann.run(input)
      prediction_success(output, expected_outputs[i]) ? success += 1 : failure += 1
      errsum += mean_squared_error(output, expected_outputs[i])
    end
    avg_mse = errsum / inputs.length.to_f

    error_rate = ((failure / inputs.length.to_f) * 100).round
    
    puts "Test results: #{success}, failure: #{failure} (classification error: #{error_rate}%, mse: #{(avg_mse * 100).round(2)}%)"

    return error_rate
  end

  private

  def self.prediction_success(actual, ideal)
    decode_output(actual) == decode_output(ideal)
  end

  def self.mean_squared_error(actual, ideal)
    errors = actual.zip(ideal).map {|a, i| a - i }
    (errors.inject(0) {|sum, err| sum += err**2}) / errors.length.to_f
  end

  def self.decode_output(output)
    (0..9).max_by {|i| output[i]}
  end

end

