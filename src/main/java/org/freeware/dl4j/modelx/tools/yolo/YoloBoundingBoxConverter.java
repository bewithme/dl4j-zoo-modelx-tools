package org.freeware.dl4j.modelx.tools.yolo;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.dom4j.Document;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;

import lombok.extern.slf4j.Slf4j;

/**
 * This class is used for converting 
 * VOC Annotation file to Yolo2BoundingBox and then we 
 * need Yolo2BoundingBox to calculate boundingBoxPriors 
 * with kmeans algorithm for yolo2 training.
 * 
 * @author xuwenfeng
 *
 */
@Slf4j
public class YoloBoundingBoxConverter {
	
	
	public static final String BOX_SIZE="size";
	public static final String IMAGE_WIDTH="width";
	public static final String IMAGE_HEIGHT="height";
	public static final String ELEMENT_OBJECT="object";
	public static final String ELEMENT_NAME="name";
	public static final String BNDBOX="bndbox";
	public static final String XMIN="xmin";
	public static final String YMIN="ymin";
	public static final String XMAX="xmax";
	public static final String YMAX="ymax";
	public static final double    GRID_SIZE=13.0d;

	
	/**
	 * convert all VOC annotation files which under specific
	 * file file directory to Yolo2BoundingBox list
	 * @param vocAnnotationsPath
	 * @return
	 * @throws Exception
	 */
	public static List<YoloBoundingBox> convertAll(String vocAnnotationsPath) throws Exception{
		
		File dir=new File(vocAnnotationsPath);
		
		if(!dir.isDirectory()) {
		
			throw new RuntimeException("vocAnnotationsPath is not Directory");
		}
	    File[]	fileArray=getSortedFiles(dir);
	    
	    List<YoloBoundingBox> retList=new ArrayList<YoloBoundingBox>(10);
	    
	    for(File file:fileArray) {
	    	
	    	  if(!file.getName().startsWith("._")&&file.getName().endsWith(".xml")) {
	    		 
	   	        retList.addAll(convert(file));
	   	      
	    	  }
	    }
		return retList;
	}
	
	
	 public static Map<String, List<YoloBoundingBox> > convertToFileMap(List<YoloBoundingBox>  list) throws Exception{
			
		 Map<String, List<YoloBoundingBox> > fileMap=new HashMap<String, List<YoloBoundingBox>>();
		 
		 
		 for(YoloBoundingBox yoloBoundingBox :list) {
			 
			 String fileName= yoloBoundingBox.getFileName();
			 
			 if(fileMap.get(fileName)==null) {
				 
				 List<YoloBoundingBox> newList=new ArrayList<YoloBoundingBox>();
				 
				 newList.add(yoloBoundingBox);
				 
				 fileMap.put(fileName,newList);
				 
			 }else {
				 
				 fileMap.get(fileName).add(yoloBoundingBox);
			 }
			
		 }	
		   
		 return fileMap;
	 }
	 
	 
	 public static Set<String> distinctLabelNames(List<YoloBoundingBox>  list) throws Exception{
			
		 Set<String> labelCounter=new HashSet<String>();
		 
		 for(YoloBoundingBox yoloBoundingBox :list) {
			 
			 labelCounter.add(yoloBoundingBox.getLabel());
			 
		 }	
		   
		 return labelCounter;
	 }
	 
	 public static void updateLabelIndex(List<YoloBoundingBox>  list, Set<String> labelNameSet) throws Exception{
			
		 for(YoloBoundingBox yoloBoundingBox :list) {
			
			 int labelIndex=getLabelIndex(labelNameSet, yoloBoundingBox.getLabel());
			 
			 yoloBoundingBox.setLabelIndex(labelIndex);
		 }
	 }
	 
	 private static int getLabelIndex(Set<String> labelNameSet,String labelNameToIndex) {
		 
		 int i=0;
		 
		 for(String labelName:labelNameSet) {
			 
			 if(labelName.equals(labelNameToIndex)) {
				return i; 
			 }
			 
			 i=i+1;
		 }
		 return -1;
	 }
	
	
	
	@SuppressWarnings("unchecked")
	private static File[] getSortedFiles(File file) {
        
        File[] files = file.listFiles();
        @SuppressWarnings("rawtypes")
		List fileList = Arrays.asList(files);
        Collections.sort(fileList, new Comparator<File>() {
            @Override
            public int compare(File o1, File o2) {
                if (o1.isDirectory() && o2.isFile())
                    return -1;
                if (o1.isFile() && o2.isDirectory())
                    return 1;
                return o1.getName().compareTo(o2.getName());
            }
        });
        
        return files;
    }
	
	/**
	 * convert single VOC annotation file
	 * to Yolo2BoundingBox list
	 * @param file
	 * @return
	 * @throws Exception
	 */
	@SuppressWarnings("unchecked")
	public static List<YoloBoundingBox> convert(File file) throws Exception {
		
		SAXReader reader = new SAXReader();
		
		Document document = reader.read(file);
		
		Element root = document.getRootElement();
		
		List<YoloBoundingBox> boundingBoxList=new ArrayList<YoloBoundingBox>(10);
		
		ImageSize imageSize=null;
		
		List<?> rootElementList = root.elements();
		
		for (int i=0;i<rootElementList.size();i++) {
			
		    Element e = (Element) rootElementList.get(i);
		    
		    String name=e.getName();
		   
		    if(BOX_SIZE.equals(name)) {
		    	
		    	imageSize=convertToImageSize(e);
		    	
		    }
		    
    	}
	    for (int i=0;i<rootElementList.size();i++) {
			
		    Element e = (Element) rootElementList.get(i);
		    
		    String name=e.getName();

		    //create one boundingBox
		    if(ELEMENT_OBJECT.equals(name)) {
		    	
		    	 VocBoundingBox boundingBox=convertToVocBoundingBox(e);
		    	 
		    	 boundingBox.setImageSize(imageSize);
		    	 
		    	 YoloBoundingBox yoloBoundingBox =convert(boundingBox);
		    	 
		    	 yoloBoundingBox.setFileName(file.getName());
		    	 
		    	 yoloBoundingBox.setBoundingBox(boundingBox);
		    	 
		    	 //filter incorrect VocBoundingBox
		    	 if(yoloBoundingBox.getH()!=0&& yoloBoundingBox.getW()!=0) {
		    		 
		    		 boundingBoxList.add(yoloBoundingBox);
		    		 
		    	 }
		    	
		    }

		}
		
		return boundingBoxList;
		
	}
	
	
	
	@SuppressWarnings("unchecked")
	public static List<VocBoundingBox> convertToVocBoundingBox(File file) throws Exception {
		
		SAXReader reader = new SAXReader();
		
		Document document = reader.read(file);
		
		Element root = document.getRootElement();
		
		List<VocBoundingBox> boundingBoxList=new ArrayList<VocBoundingBox>(10);
		
		ImageSize imageSize=null;
		
		List<?> rootElementList = root.elements();
		
		for (int i=0;i<rootElementList.size();i++) {
			
		    Element e = (Element) rootElementList.get(i);
		    
		    String name=e.getName();
		   
		    if(BOX_SIZE.equals(name)) {
		    	
		    	imageSize=convertToImageSize(e);
		    	
		    }
		    
    	}
	    for (int i=0;i<rootElementList.size();i++) {
			
		    Element e = (Element) rootElementList.get(i);
		    
		    String name=e.getName();

		    //create one boundingBox
		    if(ELEMENT_OBJECT.equals(name)) {
		    	
		    	 VocBoundingBox boundingBox=convertToVocBoundingBox(e);
		    	 
		    	 boundingBox.setImageSize(imageSize);
		    	 
		         boundingBoxList.add(boundingBox);
		    		 
		    }

		}
		
		return boundingBoxList;
		
	}
	
	
	

	/**
	 * convert xml element to ImageSize
	 * @param e
	 * @return
	 */
	private static ImageSize convertToImageSize(Element e) {
		
		ImageSize imageSize=new ImageSize();
		
		int imageWidth=0;
		
		int imageHeight=0;
		
		@SuppressWarnings("unchecked")
		List<Element> elementList=e.elements();
		 
		 for(Element element:elementList) {
			 
			 String sizeName=element.getName();
			 
			 if(IMAGE_WIDTH.equals(sizeName)) {
				   String value=element.getStringValue();
				   imageWidth=Integer.parseInt(value);
			 }
		     if(IMAGE_HEIGHT.equals(sizeName)) {
		    	   String value=element.getStringValue();
				   imageHeight=Integer.parseInt(value);
			 }
			 
		 }
		 
		 imageSize.setWidth(imageWidth);
		 
		 imageSize.setHeight(imageHeight);
		 
		 return imageSize;
	}

	/**
	 * convert xml element to VocBoundingBox
	 * @param e
	 * @return
	 */
	private static VocBoundingBox convertToVocBoundingBox(Element e) {
		
		VocBoundingBox boundingBox=new VocBoundingBox();
		
		@SuppressWarnings("unchecked")
		List<Element> elementList=e.elements();
		 
		 for(Element element:elementList) {
			 
		     String objectName=element.getName();
			 
			 if(ELEMENT_NAME.equals(objectName)) {
				 
				 String value=element.getStringValue();
				 
				 boundingBox.setLabel(value);
			 
			 }
			 
		     if(BNDBOX.equals(objectName)) {
		    	 
		    	 @SuppressWarnings("unchecked")
				List<Element> bndboxElementList=element.elements();
		    	 
		    	 for(Element bndboxElement:bndboxElementList) {
		    		 
		    		 String bndboxElementName=bndboxElement.getName();
		    		 
		    		 String value=bndboxElement.getStringValue();
		 			
		    		 Integer intValue=Integer.parseInt(value.trim());
		    		 
		    		 if(XMIN.equals(bndboxElementName)) {
		    			 boundingBox.setXmin(intValue);
		    		 }
		             if(YMIN.equals(bndboxElementName)) {
		            	 boundingBox.setYmin(intValue);
		    		 }
		    		 if(XMAX.equals(bndboxElementName)) {
		    			 boundingBox.setXmax(intValue);
		    		 }
		             if(YMAX.equals(bndboxElementName)) {
		            	 boundingBox.setYmax(intValue);
		    		 }
		    		 
		    	 }
				
		     }
			 
		 }
		 return boundingBox;
	}
	
	/**
	 * Convert VocBoundingBox to Yolo2BoundingBox
	 * 
	 * @param boundingBox
	 * @return
	 */
	public static YoloBoundingBox convert(VocBoundingBox boundingBox) {
	  
		ImageSize mageSize=boundingBox.getImageSize();
		
	    double w = boundingBox.getXmax() - boundingBox.getXmin();
	    
	    double h = boundingBox.getYmax() - boundingBox.getYmin();
	    
	    double cellW=mageSize.getWidth()/GRID_SIZE;
	    
	    double cellH=mageSize.getHeight()/GRID_SIZE;
	    
	    w=w/cellW;
	    
		h=h/cellH;
		
	    YoloBoundingBox bormalizeBoundingBox=new YoloBoundingBox();
	
	    bormalizeBoundingBox.setW(w);
	    
	    bormalizeBoundingBox.setH(h);
	    
	    bormalizeBoundingBox.setX(w/2);
	    
	    bormalizeBoundingBox.setY(h/2);
	    
	    bormalizeBoundingBox.setLabel(boundingBox.getLabel());
	    
	    return bormalizeBoundingBox;
  }
	
	
	


}
