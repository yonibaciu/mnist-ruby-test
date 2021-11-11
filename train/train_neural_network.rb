#!/usr/bin/env ruby

require 'rubygems'
require 'bundler/setup'
require 'ruby-fann' # gem install ruby-fann
require 'pry'
require './mnist_loader'
require './test_neural_network'
require './image_printer'

TRAIN_SIZE = 60_000
HIDDEN_NEURONS = [300]
MAX_TRAINING_EPOCHS = 1000
DESIRED_MEAN_SQUARED_ERROR = 0.01

puts "Loading training data..."
inputs, expected_outputs = MnistLoader.training_set.get_data_and_labels(TRAIN_SIZE)
train_data = RubyFann::TrainData.new(:inputs=> inputs, :desired_outputs=> expected_outputs)
# Print first 100 images as a char matrix to the terminal
(0...100).each do |i|
  ImagePrinter.print_image(inputs[i], expected_outputs[i])
end

fann = RubyFann::Standard.new(:num_inputs=> 24*24, :hidden_neurons=> HIDDEN_NEURONS, :num_outputs=> 10)

puts "Training network with #{inputs.length} examples..."
t = Time.now
fann.train_on_data(train_data, MAX_TRAINING_EPOCHS, 1, DESIRED_MEAN_SQUARED_ERROR)
puts "Training time: #{(Time.now - t).round(1)}s"

error_rate = TestNeuralNetwork.run(fann)

filename = "data/trained_nn_24x24_#{HIDDEN_NEURONS * '_'}_#{inputs.length}_#{error_rate}.net"
puts "Saving neural network to file: #{filename}"
fann.save(filename)

