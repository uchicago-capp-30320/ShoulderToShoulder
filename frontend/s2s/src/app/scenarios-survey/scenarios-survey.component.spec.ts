import { ComponentFixture, TestBed } from '@angular/core/testing';
import { FormsModule, ReactiveFormsModule, FormGroupDirective } from '@angular/forms';

// primeng
import { DropdownModule } from 'primeng/dropdown';
import { RadioButtonModule } from 'primeng/radiobutton';

// components and services
import { ScenariosSurveyComponent } from './scenarios-survey.component';
import { Hobby } from '../_data-models/hobby';

// helpers
import { ScenarioInterface } from '../_helpers/scenario';

describe('ScenariosSurveyComponent', () => {
  let component: ScenariosSurveyComponent;
  let fixture: ComponentFixture<ScenariosSurveyComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [
        ScenariosSurveyComponent
      ],
      imports: [
        FormsModule,
        ReactiveFormsModule,
        DropdownModule,
        RadioButtonModule
      ],
      providers: [
        FormGroupDirective
      ],
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(ScenariosSurveyComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should populate the list of scenarios with the correct number of scenarios', () => {
    expect(component.scenarios.length).toEqual(component.maxScenarios);
  });

  it('should correctly initialize the list of scenario navigation', () => {
    expect(component.scenarioNavigation.length).toEqual(component.maxScenarios);
  });

  it('should correctly update the scenario number when navigating to the next scenario', () => {
    component.nextScenario();
    expect(component.scenarioNum).toEqual(2);
  });

  it('should correctly update the scenario number when navigating to the previous scenario', () => {
    component.scenarioNum = 2;
    component.prevScenario();
    expect(component.scenarioNum).toEqual(1);
  });

  it('should correctly get a random hobby', () => {
    let hobby1: Hobby = { name: 'Arcade bar', scenarioForm: 'an arcade bar', maxParticipants: 10, type: 'GAMING' };
    let hobby2: Hobby = { name: 'Art Museums', scenarioForm: 'an art museum', maxParticipants: 10, type: 'ARTS AND CULTURE' };
    let hobby3: Hobby = { name: 'Attending Book Signing', scenarioForm: 'a book signing', maxParticipants: 5, type: 'LITERATURE' };

    component.availableHobbies = [hobby1, hobby2, hobby3];
    component.usedHobbyIndexes = [];

    let hobby = component.getHobby();
    expect(hobby).toBeTruthy();
    expect(component.usedHobbyIndexes.length).toEqual(1);
  });

  it('should correctly get an alternative for the altered variable', () => {
    let alteredVariable = 'time';
    let alternative = 'morning';
    let variableValue = 'morning';

    let newAlternative = component.getAlternative(alteredVariable, alternative, variableValue);
    expect(newAlternative).toBeTruthy();
    expect(newAlternative).not.toEqual(variableValue);
  });

  it('should get the correct class for the given scenario ID and value' , () => {
    const scenario: ScenarioInterface = {
      id: 1,
      description: 'Test scenario'
    }
    component.userService.scenariosForm.controls["scenario1"].setValue(1);
    
    let value = 1;
    let className = component.getClass(scenario, value);
    expect(className).toEqual('selected-button');

    value = 2
    className = component.getClass(scenario, value);
    expect(className).toEqual('event-button');
  });

});
