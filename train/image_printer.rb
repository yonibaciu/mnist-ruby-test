class ImagePrinter
  def self.print_image(image, labels)
    puts "------------- #{(0..9).max_by { |i| labels[i] }} -------------"
    (0...24).each do |x|
      (0...24).each do |y|
        i = x * 24 + y
        if image[i] <= 0.1
          print "  "
        elsif image[i] <= 0.5
          print " ."
        else
          print " X"
        end
      end
      puts ''
    end
    puts ''  
  end
end
