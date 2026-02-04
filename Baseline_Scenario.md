
# Performance Test Documentation: Scenario 1 (Baseline)

## 1. Project Overview
The objective of this scenario is to collect "Healthy State" (Steady State) data for the **Digital Twin** model and **MARL** (Multi-Agent Reinforcement Learning) agents. By maintaining a light, consistent load, we establish a performance baseline for resource consumption (CPU/Memory) and response times (Latency).

---

## 2. Test Configuration (Thread Group)
This section defines the volume and timing of the simulated traffic.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Number of Threads** | 20 | Simulated concurrent users hitting the system. |
| **Ramp-up Period** | 60 seconds | The time taken to reach the full 20-user load. |
| **Loop Count** | Infinite | Ensures continuous traffic until the duration expires. |
| **Duration** | 2700 seconds | Total test execution time (45 minutes). |

---

## 3. Global Settings (HTTP Request Defaults)
Centralized configuration to ensure all requests target the correct local Kubernetes entry point.
* **Protocol:** `http`
* **Server Name / IP:** `localhost`
* **Port Number:** `8080` (Robot-Shop Web/Gateway port)

---

## 4. Traffic Distribution (Random Controller)
To simulate realistic user behavior, a **Random Controller** is used. It ensures that users do not follow a predictable path, but instead pick one of the following microservice endpoints at random:

### **Microservice Endpoints (Samplers)**
1. **Web Service (Frontend):** - **Path:** `/` 
   - **Goal:** Monitor Nginx performance and UI delivery.
2. **Catalogue Service (Backend):** - **Path:** `/api/catalogue/categories` 
   - **Goal:** Trigger the Catalogue microservice and MongoDB database queries.
3. **User Service (Backend):** - **Path:** `/api/user/uniqueid` 
   - **Goal:** Trigger the User microservice and MySQL database authentication checks.

---

## 5. Load Pacing (Constant Throughput Timer)
This element acts as a "speed limit" to keep the test within **Baseline** parameters and prevent it from becoming a Stress Test.

* **Target Throughput:** $120.0$ samples per minute.
* **Effective Load:** $$\frac{120 \text{ requests}}{60 \text{ seconds}} = 2 \text{ requests per second (RPS)}$$

---

## 6. Monitoring & Verification
Two primary listeners are used to ensure test integrity:
* **View Results Tree:** Real-time verification that requests are returning `200 OK`.
* **Aggregate Report:** Tracking statistical data such as **Average Latency**, **95th Percentile**, and **Error Rate %**.

---

## 7. Execution Workflow
1. **Connectivity:** Ensure `kubectl port-forward` is active for both Prometheus (9090) and the Application (8080).
2. **Data Logging:** Start the Python Data Collection script first to capture the "Idle to Load" transition.
3. **JMeter Startup:** Execute the test plan using the "Start" (Green) button.
4. **Validation:** Periodically check the CSV file to ensure metrics like `web_cpu` and `catalogue_cpu` are being recorded with non-zero values.
