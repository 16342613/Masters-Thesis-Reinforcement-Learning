using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class FileHandler
{
    public FileHandler()
    {
        // Empty constructor
    }

    public static void WriteToFile(string filePath, List<string> lines)
    {
        try
        {
            File.AppendAllLines(filePath, lines);
        }
        catch (IOException e)
        {
            Debug.Log(e.Message);
        }
    }

    public static void WriteToFile(string filePath, string line)
    {
        try
        {
            File.AppendAllLines(filePath, new List<string>() { line });
        }
        catch (IOException e)
        {
            Debug.Log(e.Message);
        }
    }

    public static object ReadFile(string filePath, string delimiter = null)
    {
        // Read the whole text file
        string rawText = File.ReadAllText(filePath);

        // Split the string and return the parts if you want to
        if (delimiter != null)
        {
            return rawText.Split(new string[] { delimiter }, System.StringSplitOptions.None);
        }

        // Or you can return the raw text
        return rawText;
    }

    public static void ClearFile(string filePath)
    {
        if (filePath.EndsWith(".txt") == false)
        {
            throw new System.Exception("WARNING: Don't clear anything which isn't a text file!");
        }
        else
        {
            File.WriteAllText(filePath, String.Empty);
        }
    }
}
