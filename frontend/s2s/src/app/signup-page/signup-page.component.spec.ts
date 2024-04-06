import { ComponentFixture, TestBed, waitForAsync } from '@angular/core/testing';

import { SignupPageComponent } from './signup-page.component';
import { ReactiveFormsModule } from '@angular/forms';
import { FooterComponent } from '../footer/footer.component';
import { ActivatedRoute, Router } from '@angular/router';
import { InputTextModule } from 'primeng/inputtext';
import { RouterModule } from '@angular/router';


describe('SignupPageComponent', () => {
  let component: SignupPageComponent;
  let fixture: ComponentFixture<SignupPageComponent>;
  let router: Router;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [SignupPageComponent, FooterComponent],
      imports: [ReactiveFormsModule, InputTextModule, RouterModule],
      providers: [
        { provide: ActivatedRoute, useValue: {} } // Mock ActivatedRoute without any specific data
      ]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(SignupPageComponent);
    component = fixture.componentInstance;
    router = TestBed.inject(Router);
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should initialize the form correctly', () => {
    const form = component.signupForm;
    expect(form).toBeTruthy();
    expect(form.get('firstName')).toBeTruthy();
    expect(form.get('lastName')).toBeTruthy();
    expect(form.get('phoneNumber')).toBeTruthy();
    expect(form.get('email')).toBeTruthy();
    expect(form.get('password')).toBeTruthy();
    expect(form.get('confirmPassword')).toBeTruthy();
  });

  it('should reset the form', () => {
    // Set sample input values
    component.signupForm.setValue({
      firstName: 'John',
      lastName: 'Doe',
      phoneNumber: '1234567890',
      email: 'john.doe@gmail.com',
      password: 'password',
      confirmPassword: 'password'
    });

    // Reset the form
    component.resetForm();

    // Assert form is reset
    expect(component.signupForm.value).toEqual({
      firstName: null,
      lastName: null,
      phoneNumber: null,
      email: null,
      password: null,
      confirmPassword: null
    });
  });

  it('should be a valid form with valid input', () => {
    // Set sample input values
    component.signupForm.setValue({
      firstName: 'John',
      lastName: 'Doe',
      phoneNumber: '1234567890',
      email: 'john.doe@gmail.com',
      password: 'Password123!',
      confirmPassword: 'Password123!'
    });

    // Assert form is valid
    expect(component.signupForm.valid).toBe(true);
  });

  it('should not be a valid form with invalid input - non-matching passwords', () => {
    // Set sample input values
    component.signupForm.setValue({
      firstName: 'John',
      lastName: 'Doe',
      phoneNumber: '1234567890',
      email: 'john.doe@gmail.com',
      password: 'Password123!',
      confirmPassword: 'Password123' // passwords do not match
    });

    // Assert form is valid
    expect(component.signupForm.valid).toBe(false);

    // Assert confirmPassword field has an error
    expect(component.signupForm.errors).toEqual({ PasswordNoMatch: true });
  });

  it('should not be a valid form with invalid input - incorrect phone number (len)', () => {
    // Set sample input values
    component.signupForm.setValue({
      firstName: 'John',
      lastName: 'Doe',
      phoneNumber: '123456789', //length is less than 10
      email: 'john.doe@gmail.com',
      password: 'Password123!',
      confirmPassword: 'Password123!'
    });

    // Assert form is valid
    expect(component.signupForm.valid).toBe(false);

    // Assert phoneNumber field has an error
    const phoneNumberField = component.signupForm.get('phoneNumber');
    expect(phoneNumberField?.errors).toEqual({ minlength: { requiredLength: 10, actualLength: 9 } });
  });

  it('should not be a valid form with invalid input - incorrect phone number (ints only)', () => {
    // Set sample input values
    component.signupForm.setValue({
      firstName: 'John',
      lastName: 'Doe',
      phoneNumber: '123456789!', //length is less than 10
      email: 'john.doe@gmail.com',
      password: 'Password123!',
      confirmPassword: 'Password123!'
    });

    // Assert form is valid
    expect(component.signupForm.valid).toBe(false);

    // Assert phoneNumber field has an error
    const phoneNumberField = component.signupForm.get('phoneNumber');
    expect(phoneNumberField?.errors).toEqual({ pattern: { requiredPattern: '^[0-9]*$', actualValue: '123456789!' } });
  });

  it('should not be a valid form with invalid input - incorrect email', () => {
    // Set sample input values
    component.signupForm.setValue({
      firstName: 'John',
      lastName: 'Doe',
      phoneNumber: '1234567890',
      email: 'john.doegmail.com', //missing @
      password: 'Password123!',
      confirmPassword: 'Password123!'
    });

    // Assert form is invalid
    expect(component.signupForm.valid).toBe(false);

    // Assert email field has an error
    const emailField = component.signupForm.get('email');
    expect(emailField?.errors).toEqual({ email: true });
  });

  it('should not be a valid form with invalid input - first name required', () => {
    // Set sample input values
    component.signupForm.setValue({
      firstName: '', // first name is required
      lastName: 'Doe',
      phoneNumber: '1234567890',
      email: 'john.doe@gmail.com',
      password: 'Password123!',
      confirmPassword: 'Password123!'
    });

    // Assert form is valid
    expect(component.signupForm.valid).toBe(false);

    // Assert firstName field has an error
    const firstNameField = component.signupForm.get('firstName');
    expect(firstNameField?.errors).toEqual({ required: true });
  });

  it('should not be a valid form with invalid input - last name required', () => {
    // Set sample input values
    component.signupForm.setValue({
      firstName: 'John',
      lastName: '', // last name is required
      phoneNumber: '1234567890',
      email: 'john.doe@gmail.com',
      password: 'Password123!',
      confirmPassword: 'Password123!'
    });

    // Assert form is valid
    expect(component.signupForm.valid).toBe(false);

    // Assert firstName field has an error
    const lastNameField = component.signupForm.get('lastName');
    expect(lastNameField?.errors).toEqual({ required: true });
  });

  it('should be a valid form with valid input - phone number optional', () => {
    // Set sample input values
    component.signupForm.setValue({
      firstName: 'John',
      lastName: 'Doe',
      phoneNumber: '',
      email: 'john.doe@gmail.com',
      password: 'Password123!',
      confirmPassword: 'Password123!'
    });

    // Assert form is valid
    expect(component.signupForm.valid).toBe(true);
  });

  it('should not be a valid form with invalid input - email required', () => {
    // Set sample input values
    component.signupForm.setValue({
      firstName: 'John',
      lastName: 'Doe',
      phoneNumber: '1234567890',
      email: '', // email required
      password: 'Password123!',
      confirmPassword: 'Password123!'
    });

    // Assert form is valid
    expect(component.signupForm.valid).toBe(false);

    // Assert firstName field has an error
    const emailField = component.signupForm.get('email');
    expect(emailField?.errors).toEqual({ required: true });
  });

  it('should not be a valid form with invalid input - password required', () => {
    // Set sample input values
    component.signupForm.setValue({
      firstName: 'John',
      lastName: 'Doe',
      phoneNumber: '1234567890',
      email: 'john.doe@gmail.com',
      password: '', // password required
      confirmPassword: 'Password123!'
    });

    // Assert form is valid
    expect(component.signupForm.valid).toBe(false);

    // Assert firstName field has an error
    const passwordField = component.signupForm.get('password');
    expect(passwordField?.errors).toEqual({ required: true });
  });

  it('should not be a valid form with invalid input - confirm password required', () => {
    // Set sample input values
    component.signupForm.setValue({
      firstName: 'John',
      lastName: 'Doe',
      phoneNumber: '1234567890',
      email: 'john.doe@gmail.com',
      password: 'Password123!',
      confirmPassword: '' // confirm password required
    });

    // Assert form is valid
    expect(component.signupForm.valid).toBe(false);

    // Assert firstName field has an error
    const confirmPasswordField = component.signupForm.get('confirmPassword');
    expect(confirmPasswordField?.errors).toEqual({ required: true });
  });

  it('should navigate to /onboarding route on successful form submission', () => {
    const navigateSpy = spyOn(router, 'navigate');
    
    // Set sample input values
    component.signupForm.setValue({
      firstName: 'John',
      lastName: 'Doe',
      phoneNumber: '1234567890',
      email: 'john.doe@gmail.com',
      password: 'Password123!',
      confirmPassword: 'Password123!'
    });

    // Trigger form submission
    component.onSubmit();

    // Expect navigation to /onboarding route
    expect(navigateSpy).toHaveBeenCalledWith(['/onboarding']);
  });
});
