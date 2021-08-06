using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.IO;

public class HelperScript : MonoBehaviour
{
    public static List<GameObject> FindChildrenWithTag(Transform parentTransform, string desiredTag)
    {
        List<GameObject> gameObjects = new List<GameObject>();

        foreach (Transform child in parentTransform)
        {
            if (child.tag == desiredTag)
            {
                gameObjects.Add(child.gameObject);
            }
        }

        return gameObjects;
    }

    public static List<float> ParseStringToFloats(string toParse, string delimiter)
    {
        List<string> stringFloats = toParse.Split(new string[] { delimiter }, StringSplitOptions.None).ToList();
        List<float> output = new List<float>();

        foreach (string stringFloat in stringFloats)
        {
            output.Add(float.Parse(stringFloat));
        }

        return output;
    }

    public static void PrintList<T>(List<T> toPrint)
    {
        Debug.Log(string.Join("   |   ", toPrint.ToArray()));
    }

    public static Vector3 GetDirection(Vector3 from, Vector3 to)
    {
        Vector3 heading = to - from;

        return heading / heading.magnitude;
    }

    public static void ClearLogs(string logFolderPath = "Assets/Debug")
    {
        foreach (string logFilePath in Directory.GetFiles(logFolderPath))
        {
            if (logFilePath.EndsWith(".txt") == true)
            {
                FileHandler.ClearFile(logFilePath);
            }
        }
    }

    public static int SampleProbabilityDistribution(List<float> probabilites)
    {
        float currentInterval = 0;
        float randomNumber = UnityEngine.Random.Range(0, 1f);

        for (int i = 0; i < probabilites.Count; i++)
        {
            currentInterval = currentInterval + probabilites[i];

            if (randomNumber <= currentInterval)
            {
                return i;
            }
        }

        Debug.Log("PROBABILITY ERROR!");
        return 0;
    }

    /*public static Vector3 NormalizeVector3(Vector3 original, float squareSide)
    {
        float length = (float)Math.Sqrt(Math.Pow(original.x, 2f) + Math.Pow(original.x, 2f) + Math.Pow(original.x, 2f));
        return new Vector3(original.x / length, original.y / length, original.z / length);
    }*/
}
