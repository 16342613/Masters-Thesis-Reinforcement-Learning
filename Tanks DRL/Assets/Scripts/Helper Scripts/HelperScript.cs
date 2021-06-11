using System.Collections;
using System.Collections.Generic;
using UnityEngine;

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
}
