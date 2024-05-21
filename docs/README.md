# ShoulderToShoulder Developer Onboarding: Instructions and Documentation

After cloning the repository to your local machine, you will need to set up the virtual environment and install necessary dependencies in order to begin developing. 

## Virtual Environments

#### `frontend`

The frontend development uses `npm` as its package manager. To use npm, you first need to install [nodejs](https://nodejs.org/en). (follow the installation instructions provided on the nodejs website, or run `brew install nodejs` in your terminal). Afterward, to install the necessary packages, follow these steps:

<pre>
```
cd frontend
npm install -g @angular/cli
npm install
```
</pre>

Once the packages have been installed, you can begin developing in `cd frontend`. 

#### `backend`

The backend employs a poetry virtual environment with Python 3.12. Enter the poetry environment every time you develop in the backend. To open the environment, follow these steps:
 
<pre>
```
cd backend
poetry env use 3.12
poetry shell
poetry install  // do this on the first entry or for any changes to the packages

#to exit the environment
exit
```
</pre>


## How To Run the App

First, make sure you have successfully completed the set up of the virtual environments (see instructions above).

To open and use the application, you will need to launch both the frontend module and the backend module at the same time in order to get the frontend and backend working and commmunicating in tandem. Follow these steps in the terminal:

<pre>
```
cd frontend/s2s
ng serve
```
</pre>

Navigate to `localhost:4200/` in your web browser. 

Now, open a new terminal and run:

<pre>
```
cd backend
poetry env use 3.12
poetry shell
python shoulder/manage.py runserver
```
</pre>

Navigate to `localhost:1800/data` in your web browser and enter the superuser log-in credentials.

To exit the application, run ctrl+C (i.e. ^C) in both terminals to shut down the local hosts.

## Running The App On A Server

<!-- TBD -->

## Unit Testing

Following development in any module, run it's unit testing. 

#### `frontend`

<!-- TBD -->

<pre>
```

```
</pre>

#### `backend`
 
<pre>
```
cd backend
poetry env use 3.12
poetry shell
poetry install
cd shoulder

pytest tests    // All of the backend tests are contained in this directory.

#to exit the environment
exit
```
</pre>


## Pre-Commit Checklist

Before merging your code, it needs to be properly formatted. Follow these steps (for each development module) to pass pre-commit.

#### `backend`: 

<pre>
```
cd backend
pre-commit run --all
```
</pre>

#### `frontend`: 

<pre>
```
cd frontend
npm run lint
npm run format
```
</pre>
