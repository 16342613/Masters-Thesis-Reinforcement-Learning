using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

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

        foreach(string stringFloat in stringFloats)
        {
            output.Add(float.Parse(stringFloat));
        }

        return output;
    }

    public static void PrintList<T>(List<T> toPrint)
    {
        Debug.Log(string.Join(" | ", toPrint.ToArray()));
    }
}
