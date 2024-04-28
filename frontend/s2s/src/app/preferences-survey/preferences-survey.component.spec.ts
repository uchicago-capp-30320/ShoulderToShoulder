import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpClientModule } from '@angular/common/http';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { FormsModule, ReactiveFormsModule, FormGroupDirective } from '@angular/forms';

// primeng
import { DropdownModule } from 'primeng/dropdown';
import { MultiSelectModule } from 'primeng/multiselect';
import { TooltipModule } from 'primeng/tooltip';

// components and services
import { PreferencesSurveyComponent } from './preferences-survey.component';

describe('PreferencesSurveyComponent', () => {
  let component: PreferencesSurveyComponent;
  let fixture: ComponentFixture<PreferencesSurveyComponent>;
  let httpController: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [
        PreferencesSurveyComponent
      ],
      imports: [
        HttpClientModule,
        DropdownModule,
        MultiSelectModule,
        FormsModule,
        ReactiveFormsModule,
        TooltipModule,
        HttpClientTestingModule
      ],
      providers: [
        FormGroupDirective
      ]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(PreferencesSurveyComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
    httpController = TestBed.inject(HttpTestingController);
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should create a string list of hobbies', () => {
    expect(component.hobbies).toBeTruthy();
    expect(component.hobbies.length).toBeGreaterThan(0);
    expect(component.leastInterestedHobbies).toBeTruthy();
    expect(component.leastInterestedHobbies.length).toBeGreaterThan(0);
    expect(component.mostInterestedHobbies).toBeTruthy();
    expect(component.mostInterestedHobbies.length).toBeGreaterThan(0);
  });

  it('should make the least interested hobbies different from the most interested hobbies', () => {
    component.mostInterestedHobbies = ['hobby1', 'hobby2', 'hobby3'];
    component.leastInterestedHobbies = ['hobby1', 'hobby2', 'hobby3'];
    component.hobbies = ['hobby1', 'hobby2', 'hobby3'];

    expect(component.leastInterestedHobbies).toEqual(component.mostInterestedHobbies);

    component.onboardingService.preferencesForm.setValue({
      leastInterestedHobbies: ['hobby1'],
      mostInterestedHobbies: ['hobby2'],
      zipCode: '12345',
      city: 'city',
      state: 'state',
      addressLine1: "addressLine1",
      groupSizes: ['groupSizes'],
      eventFrequency: 'eventFrequency',
      eventNotifications: 'eventNotifications',
      distances: 'distances'
    });

    fixture.detectChanges();
    component.getLeastInterestedHobbiesArray();
    expect(component.leastInterestedHobbies).not.toEqual(component.mostInterestedHobbies);
  });
});
