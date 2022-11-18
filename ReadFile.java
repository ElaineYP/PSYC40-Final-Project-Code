import java.util.*;
import java.io.File;  // Import the File class
import java.io.FileNotFoundException;  // Import this class to handle errors
import java.util.Scanner; // Import the Scanner class to read text files

public class ReadFile {
  private File fp;
  private String fileName;
  private ArrayList<Double> ls = new ArrayList<>();
  private Scanner reader;

  public ReadFile (String fileName){
    this.fileName = fileName;
    fp = new File (this.fileName);
  }

  public ArrayList<Double> fileToList(){
    try{
    reader = new Scanner(fp);
    while (reader.hasNextLine()) {
        String line = reader.nextLine();
        double data = Double.valueOf(line);
        //debug
        //System.out.println(data);
        ls.add(data);
      }
    }
    catch (FileNotFoundException e) {
        System.out.println("file not found");
        e.printStackTrace();
      }
    return ls;
  }
}
