/*    
*    Main.java 
*    Copyright (C) 2017 Universidade Federal do Paran�, Curitiba, Paran�, Brasil
*    @Author Paulo Ricardo Lisboa de Almeida (prlalmeida@inf.ufpr.br)
*    This program is free software: you can redistribute it and/or modify
*    it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*    
*    This program is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU General Public License for more details.
*    
*    You should have received a copy of the GNU General Public License
*    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
package br.ufpr.dynse;

import br.ufpr.dynse.testbed.MultipleExecutionsTestbed;
import br.ufpr.dynse.testbed.MaskedTestBed;

public class Main {
    public static void main( String[] args ) {
		System.out.println(String.join(" ",args));
    	String inputFilePath = args[2];
    	String outputFileName = args[3];
		try{
			MaskedTestBed testBed = new MaskedTestBed();
			testBed.executeExternalTests(1, inputFilePath, outputFileName);
		}catch(Exception e){
			e.printStackTrace();
		}
		System.out.println("Done!");
    }
}