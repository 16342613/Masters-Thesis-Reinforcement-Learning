using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class MasterTrainerScript : MonoBehaviour
{
    List<GameObject> environments = new List<GameObject>();
    List<MovementTrainerAI> trainingScripts = new List<MovementTrainerAI>();

    public int environmentCount;
    public GameObject environment;
    public Vector3 offset = new Vector3(0, 0, 10);

    // Start is called before the first frame update
    void Start()
    {
        SetUpEnvironments();

        environments = GameObject.FindGameObjectsWithTag("Environment").ToList();
        foreach (GameObject environment in environments)
        {
            trainingScripts.Add(environment.GetComponentInChildren<MovementTrainerAI>());
        }

        for (int i=0; i<trainingScripts.Count; i++)
        {
            trainingScripts[i].masterTrainer = this;
            trainingScripts[i].AIName = i.ToString();
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.M))
        {
            foreach (MovementTrainerAI trainingScript in trainingScripts)
            {
                trainingScript.TestConnection();
            }
        }

        if (Input.GetKeyDown(KeyCode.B))
        {
            Train();
        }

        // Debug.Log(GlobalScript.globalEpisodeCount);
    }

    private void SetUpEnvironments()
    {
        Vector3 currentSpawnPoint = Vector3.zero;

        for (int i = 0; i < environmentCount; i++)
        {
            GameObject newEnvironment = GameObject.Instantiate(environment);
            newEnvironment.transform.position = currentSpawnPoint;

            currentSpawnPoint += offset;
        }
    }

    private void Train()
    {
        foreach(MovementTrainerAI trainingScript in trainingScripts)
        {
            StartCoroutine(trainingScript.TrainA3C());
        }
    }
}
