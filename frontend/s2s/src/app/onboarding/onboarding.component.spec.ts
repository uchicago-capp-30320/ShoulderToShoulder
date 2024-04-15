import { ComponentFixture, TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';
import { FormsModule, ReactiveFormsModule, FormControlDirective } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';

// primeng
import { ButtonModule } from 'primeng/button';
import { DropdownModule } from 'primeng/dropdown';
import { MultiSelectModule } from 'primeng/multiselect';
import { TooltipModule } from 'primeng/tooltip';
import { CheckboxModule } from 'primeng/checkbox';
import { RadioButton, RadioButtonModule } from 'primeng/radiobutton';

// components and services
import { OnboardingComponent } from './onboarding.component';
import { FooterComponent } from '../footer/footer.component';
import { DemographicsSurveyComponent } from '../demographics-survey/demographics-survey.component';
import { PreferencesSurveyComponent } from '../preferences-survey/preferences-survey.component';
import { EventAvailabilitySurveyComponent } from '../event-availability-survey/event-availability-survey.component';
import { ScenariosSurveyComponent } from '../scenarios-survey/scenarios-survey.component';


describe('OnboardingComponent', () => {
  let component: OnboardingComponent;
  let fixture: ComponentFixture<OnboardingComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [
        OnboardingComponent, 
        FooterComponent, 
        DemographicsSurveyComponent,
        PreferencesSurveyComponent,
        EventAvailabilitySurveyComponent,
        ScenariosSurveyComponent
      ],
      imports: [
        ButtonModule,
        HttpClientModule,
        DropdownModule,
        MultiSelectModule,
        FormsModule,
        ReactiveFormsModule,
        TooltipModule,
        CheckboxModule,
        RadioButtonModule
      ],
      providers: [
        FormControlDirective
      ]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(OnboardingComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should initialize the page number correctly', () => {
    expect(component.page).toBe(1);
  });

  it('should correctly increment the page number', () => {
    component.nextPage();
    expect(component.page).toBe(2);
  });

  it('should correctly decrement the page number', () => {
    component.page = 2;
    component.previousPage();
    expect(component.page).toBe(1);
  });

  it('should disable the next button when the user preferences form is invalid and page is 3', () => {
    component.page = 3;
    component.userService.preferencesForm.setErrors({ invalid: true });
    fixture.detectChanges();
    const button = fixture.debugElement.query(By.css('#next-button')).nativeElement;
    expect(button.disabled).toBeTrue(); // Using toBeTrue for better semantics
  });

  it('should disable the submit button when the user scenario form is invalid and page is 5', () => {
    component.page = 5;
    component.userService.scenariosForm.setErrors({ invalid: true });
    fixture.detectChanges();
    const button = fixture.debugElement.query(By.css('#submit-button')).nativeElement;
    expect(button.disabled).toBeTrue(); // Using toBeTrue for better semantics
  });
});
